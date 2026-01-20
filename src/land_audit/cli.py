"""Command-line interface for land audit."""

import logging
import sys
from pathlib import Path

import click

from land_audit.audit import LandAudit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--lat",
    type=float,
    help="Latitude for point mode",
)
@click.option(
    "--lon",
    type=float,
    help="Longitude for point mode",
)
@click.option(
    "--radius-m",
    type=float,
    default=2000,
    help="Radius in meters for point mode (default: 2000)",
)
@click.option(
    "--geojson",
    type=click.Path(exists=True, path_type=Path),
    help="Path to GeoJSON file for polygon mode",
)
@click.option(
    "--cameras-n",
    type=int,
    default=6,
    help="Total number of cameras to distribute across zones (default: 6)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default="out",
    help="Output directory (default: out/)",
)
@click.option(
    "--dem-resolution-m",
    type=int,
    default=10,
    help="DEM resolution in meters (default: 10, options: 10 or 30)",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    help="Cache directory for DEM files (default: data/cache)",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    lat,
    lon,
    radius_m,
    geojson,
    cameras_n,
    output_dir,
    dem_resolution_m,
    cache_dir,
    verbose,
):
    """Orome Land Audit Engine - Generate camera placement recommendation zones.

    Run in one of two modes:

    \b
    Point mode:
      land-audit --lat 42.45 --lon -76.48 --radius-m 2000 --cameras-n 6

    \b
    Polygon mode:
      land-audit --geojson property.geojson --cameras-n 8

    Generates priority zones for camera placement rather than exact coordinates,
    allowing for on-the-ground judgment based on vegetation, access, and field conditions.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input mode
    if geojson:
        if lat or lon:
            click.echo("Error: Cannot specify both --geojson and --lat/--lon", err=True)
            sys.exit(1)
        mode = "polygon"
    elif lat and lon:
        mode = "point"
    else:
        click.echo("Error: Must specify either --geojson OR --lat and --lon", err=True)
        sys.exit(1)

    try:
        # Initialize audit
        audit = LandAudit(output_dir=output_dir, cache_dir=cache_dir)

        # Run audit
        click.echo(f"🌲 Starting land audit in {mode} mode...")
        click.echo(f"   Output directory: {output_dir}")
        click.echo(f"   Total cameras to distribute: {cameras_n}")
        click.echo(f"   DEM resolution: {dem_resolution_m}m")
        click.echo(f"   Strategy: Zone-based recommendations (not exact placements)")
        click.echo()

        if mode == "point":
            report_path = audit.run_point_audit(
                lat=lat,
                lon=lon,
                radius_m=radius_m,
                cameras_n=cameras_n,
                dem_resolution_m=dem_resolution_m,
            )
        else:  # polygon
            report_path = audit.run_polygon_audit(
                geojson_path=geojson,
                cameras_n=cameras_n,
                dem_resolution_m=dem_resolution_m,
            )

        click.echo()
        click.echo("✅ Land audit complete!")
        click.echo(f"   📄 Report: {report_path}")
        click.echo(f"   🗺️  Recommendation Zones: {output_dir / 'audit.geojson'}")
        click.echo(f"   🔍 Heat Maps & Analysis: {output_dir / 'debug/'}")
        click.echo(f"   🌐 3D Visualization: {output_dir / 'terrain_3d.html'}")
        click.echo()
        click.echo("💡 Next steps:")
        click.echo("   • Review priority zones in the PDF report")
        click.echo("   • View 3D terrain map with heat maps")
        click.echo("   • Place cameras within zones using field judgment")

    except Exception as e:
        logger.exception("Audit failed")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
